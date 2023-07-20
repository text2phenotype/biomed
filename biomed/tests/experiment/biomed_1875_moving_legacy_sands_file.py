"""
The process for running this was not quite a single script but is worth documenting:


The problem was caused by old sands formatting of storing files at the root within a bucket and this file format
being incompatible within the async processsing framework we are working with

Step 1: Get list of all uuids you want to include (space seperated)

a) I ran the following command in my terminal:


 aws s3 sync s3://us-west-2 raw_text --exclude '*' --include '*.txt' --exclude '*text*' --exclude 's3://mdl-phi-cyan-us-west-2/processed/*' --exclude '*redacted*' --exclude 's3://us-west-2/corpus/*' --exclude 's3://us-west-2/job6/*' --exclude '*corpus*'

Which copied all base files into /Users/shannon.fee/Downloads/raw_text

b) ls /Users/shannon.fee/Downloads/raw_text

c) copy output of above terminal command into python terminal


"""
a="""033c8232-8ef2-4ef0-9d9d-92a89990e536.txt		4577e9bf-229a-40f9-81f2-fc74b151e513.txt		7dbd0223-897e-4e16-89ed-a54a055d1308.txt		b53846fb-8956-40cd-bf5f-001e9306dbb2_redacted.txt
0521fad5-92ee-4384-8d11-7b0190963cb9.txt		45823bb3-0fed-457b-84ec-c93ed518dfee.txt		7dbd0223-897e-4e16-89ed-a54a055d1308_redacted.txt	b8f4b849-10e8-4224-a0c5-dcecce718b88.txt
0521fad5-92ee-4384-8d11-7b0190963cb9_redacted.txt	4ae0ad9b-ea4e-40f5-964f-f58727cbeb85.txt		7dcf2c9d-6caa-4688-b86f-f49fc1c7ad0c.txt		bc1c9e2a-8431-49db-933a-7fad81afa9d3.txt
05656cef-77f2-4d26-8e8d-26806f83ae83.txt		4b9adaad-3e15-4100-95d9-4b646893016d.txt		7f6c97d6-fcac-496c-9a56-f47657154356.txt		bc1c9e2a-8431-49db-933a-7fad81afa9d3_redacted.txt
05b77a65-28b2-4e58-90e2-8a0258e6ba5c.txt		4b9adaad-3e15-4100-95d9-4b646893016d_redacted.txt	7f6c97d6-fcac-496c-9a56-f47657154356_redacted.txt	bd8fd756-9062-4eb1-a87e-71e9c999b6f2.txt
061ae1d6-a249-428d-bebf-e5e600e0725e.txt		4bff4f3a-3a53-4334-8574-26d542ea4ff1.txt		801868fd-75be-4720-8501-c04a3b71c829.txt		be4b76a7-0ab4-4a43-990e-74396f187759.txt
061ae1d6-a249-428d-bebf-e5e600e0725e_redacted.txt	4c3c8691-ba8c-4d64-9a9c-2fb19aff7488.txt		801868fd-75be-4720-8501-c04a3b71c829_redacted.txt	becbf847-4e55-45b0-adeb-c2fa9c647f2a.txt
07421df4-fe2c-4c91-841a-98b0b4defe62.txt		4c3c8691-ba8c-4d64-9a9c-2fb19aff7488_redacted.txt	81f538eb-4bbb-456d-b82e-f02f40ff409b.txt		becbf847-4e55-45b0-adeb-c2fa9c647f2a_redacted.txt
07caf45f-d73f-406c-ade5-d9fce3374288.txt		4cfebc1a-5ffb-4280-859e-ded2579d43b0.txt		81f538eb-4bbb-456d-b82e-f02f40ff409b_redacted.txt	bfa2b78d-193d-4cc5-bcdf-54c980e67edb.txt
07caf45f-d73f-406c-ade5-d9fce3374288_redacted.txt	4cfebc1a-5ffb-4280-859e-ded2579d43b0_redacted.txt	8383debd-3875-488e-9661-991f6f8f3cae.txt		c0c6652f-551b-41dd-a08d-2cc20e8cc24c.txt
07ec6036-7bec-47b9-bb92-6f80409c7d81.txt		4da3c709-8447-4486-9c36-f9162271ed5e.txt		83f63eb6-4f8e-4f1d-b118-17e10fdab76e.txt		c185d0a1-2a37-4b09-b271-e13f76e1c6b9.txt
08a60da4-344d-4954-be1a-100599857a54.txt		4dc62c5c-8fd9-4214-adc1-a09eb70d8686.txt		83f63eb6-4f8e-4f1d-b118-17e10fdab76e_redacted.txt	c185d0a1-2a37-4b09-b271-e13f76e1c6b9_redacted.txt
09737523-ecb3-4d98-af41-d96be0c38225.txt		4dc62c5c-8fd9-4214-adc1-a09eb70d8686_redacted.txt	8503afa8-60a7-45ea-aefc-a7093a91dc66.txt		c21b5045-c0c3-479f-bcfa-f769410f6a2e.txt
09737523-ecb3-4d98-af41-d96be0c38225_redacted.txt	4dfb8f32-0917-47e6-a66e-b54a41a88246.txt		8503afa8-60a7-45ea-aefc-a7093a91dc66_redacted.txt	c2631971-8691-4842-8d52-1f4559b4b6bd.txt
0a889f47-1dff-422b-b693-fba1d89d2eb5.txt		4e45152f-1096-4e84-baa1-853a6891e6da.txt		866a5da0-bfcb-4aea-85dc-fefb3696915d.txt		c34c26e7-e9bb-4388-ab49-a0c4f5ec93c3.txt
0a889f47-1dff-422b-b693-fba1d89d2eb5_redacted.txt	4e45152f-1096-4e84-baa1-853a6891e6da_redacted.txt	866a5da0-bfcb-4aea-85dc-fefb3696915d_redacted.txt	c3f3d771-f1ca-4674-ad9f-33bf795781e9.txt
0efb8306-44c1-45ee-864c-4c004a94a4cd.txt		50a1d08a-f5a2-4e9a-b077-610f20b19498.txt		86ccb927-e78f-4901-9834-44d7459d56d8.txt		c3f3d771-f1ca-4674-ad9f-33bf795781e9_redacted.txt
0efb8306-44c1-45ee-864c-4c004a94a4cd_redacted.txt	5199c793-09bf-4dbe-8db4-08df7f293705.txt		86ccb927-e78f-4901-9834-44d7459d56d8_redacted.txt	c4ab2ac5-b7a0-4066-9398-c44c96035962.txt
0fe79f6f-aa3c-4248-8ee5-f074e35bf1c5.txt		5199c793-09bf-4dbe-8db4-08df7f293705_redacted.txt	88f7d368-6edf-4039-a5f0-e3fd5a16c3da.txt		c4ab2ac5-b7a0-4066-9398-c44c96035962_redacted.txt
0fe79f6f-aa3c-4248-8ee5-f074e35bf1c5_redacted.txt	5277dae6-31ea-44af-9cdc-33ccd279350e.txt		88f7d368-6edf-4039-a5f0-e3fd5a16c3da_redacted.txt	c51c7dd7-48d5-4daf-b61d-0b83527e37e7.txt
10253cc8-a2f9-4515-8891-cd9369a5593c.txt		5277dae6-31ea-44af-9cdc-33ccd279350e_redacted.txt	8ac22b2d-ca18-406b-9fbb-bd0cb64aee42.txt		c53990cf-abf0-438c-9066-7ae7f773c492.txt
10253cc8-a2f9-4515-8891-cd9369a5593c_redacted.txt	57842fa1-9686-4338-923e-179c01232053.txt		8ac22b2d-ca18-406b-9fbb-bd0cb64aee42_redacted.txt	c53990cf-abf0-438c-9066-7ae7f773c492_redacted.txt
109d9f47-fb70-40e5-b2a8-e6818586719b.txt		58b71419-9ee8-4482-ac8c-74fa4c75bc47.txt		8bd5f66e-00a1-4e73-8163-e569f694f15c.txt		c64fcdb1-da43-448e-9643-20354d139a9c.txt
109d9f47-fb70-40e5-b2a8-e6818586719b_redacted.txt	58b71419-9ee8-4482-ac8c-74fa4c75bc47_redacted.txt	8e7bf6c8-40c6-4c6c-9a95-d4d5114c7b68.txt		c7801afd-6a0f-4c8b-a803-2cedc5b2a245.txt
113da654-8002-44db-aff4-a4961c20ad02.txt		595bdd40-8af1-4c0d-943f-594016ce32e5.txt		8f19b622-13f4-402f-bd8f-ed2519e6eb1a.txt		c7801afd-6a0f-4c8b-a803-2cedc5b2a245_redacted.txt
113da654-8002-44db-aff4-a4961c20ad02_redacted.txt	595bdd40-8af1-4c0d-943f-594016ce32e5_redacted.txt	8f19b622-13f4-402f-bd8f-ed2519e6eb1a_redacted.txt	c8d1ebee-8c1e-46cf-9327-0e7511579463.txt
11db182c-d2db-4c91-9b7a-1f0926800f14.txt		595eaa1e-e614-4605-9d94-04df4f2a8471.txt		907a4c1b-892a-413f-924d-686488b64284.txt		c8d1ebee-8c1e-46cf-9327-0e7511579463_redacted.txt
121ce082-b7f2-46e4-b6f0-300dffde0b14.txt		59a4e5b4-e18d-4189-ad9e-a8eab3118537.txt		9095f60a-c07a-4c3f-951d-720428c1a2f7.txt		ca0caf6f-bc19-46cd-9feb-164f51d8fbeb.txt
121ce082-b7f2-46e4-b6f0-300dffde0b14_redacted.txt	59a4e5b4-e18d-4189-ad9e-a8eab3118537_redacted.txt	9095f60a-c07a-4c3f-951d-720428c1a2f7_redacted.txt	cb190399-7716-42fc-a7c5-4a3d039bc9c7.txt
12cecd17-c97f-41ed-bc9f-2c27b6febb09.txt		5a3c229e-0c63-4b82-90f8-dd97859fb0db.txt		90ed51eb-123b-4228-8747-dd252408b212.txt		cb190399-7716-42fc-a7c5-4a3d039bc9c7_redacted.txt
12cecd17-c97f-41ed-bc9f-2c27b6febb09_redacted.txt	5a3c229e-0c63-4b82-90f8-dd97859fb0db_redacted.txt	90ed51eb-123b-4228-8747-dd252408b212_redacted.txt	ce7fe25f-b60d-41b9-883c-102dc01a439e.txt
13ef44fc-62cf-4b5e-aac5-c7ebbfe8944a.txt		5a8184eb-0753-4b9f-a89e-fec320f2b53a.txt		91082884-5dbb-407c-a9e4-6ef26e135806.txt		ce7fe25f-b60d-41b9-883c-102dc01a439e_redacted.txt
1478c0e7-6ada-41bc-b5a9-a87aefc23c6c.txt		5baede87-1470-4660-8b35-176412d19083.txt		91d7b339-406b-4376-b234-00397efdb03e.txt		d1f02175-565e-48f5-bbee-6fe8bbb9027a.txt
149e220d-5e5d-4b49-970e-f5b4f0b2467d.txt		5c4415ae-a895-4970-bdb9-29474716a0bd.txt		9225a166-3aad-4564-a1f2-068cc242e398.txt		d1f02175-565e-48f5-bbee-6fe8bbb9027a_redacted.txt
149e220d-5e5d-4b49-970e-f5b4f0b2467d_redacted.txt	5c4415ae-a895-4970-bdb9-29474716a0bd_redacted.txt	9225a166-3aad-4564-a1f2-068cc242e398_redacted.txt	d3bb2bfb-ac1e-4c3b-a150-67b8249463ec.txt
1527ad41-60cc-47fe-a35a-02dbd1841c6a.txt		5c7afa9b-dc99-4b7a-ae1d-1719a27c4043.txt		92e6b246-1d01-4add-880d-be19bac12acb.txt		d3f42974-af0c-432e-9730-3a716947a33c.txt
154c6e9b-a6bd-4669-a3d1-6faac35eb41a.txt		5c7afa9b-dc99-4b7a-ae1d-1719a27c4043_redacted.txt	946ac89f-4488-4678-9c16-29d7e495e113.txt		d877b755-7ce8-4953-84ec-38e524d453c8.txt
154c6e9b-a6bd-4669-a3d1-6faac35eb41a_redacted.txt	5d0e6f09-cc5b-4be3-9eb7-6567058194af.txt		946ac89f-4488-4678-9c16-29d7e495e113_redacted.txt	d877b755-7ce8-4953-84ec-38e524d453c8_redacted.txt
1598de7f-1045-4113-b57e-30f27ea8318c.txt		5f21f8f5-9045-4d33-8d89-8dcd9039a308.txt		953f0fd3-29d4-463a-9b9b-555f06b9e988.txt		d90aadf2-a26d-4549-8e91-ac5b2e135e07.txt
15d1d237-20eb-4a45-adb7-04908780c06d.txt		6129d769-2045-4758-893e-0e548bd7d7f4.txt		953f0fd3-29d4-463a-9b9b-555f06b9e988_redacted.txt	d90aadf2-a26d-4549-8e91-ac5b2e135e07_redacted.txt
16162424-9311-46f1-94a4-8e2e4862f31e.txt		6129d769-2045-4758-893e-0e548bd7d7f4_redacted.txt	9583ff38-030d-4d74-9849-2e047285b32f.txt		da103ce1-908a-45a4-ae09-999754a00a92.txt
16162424-9311-46f1-94a4-8e2e4862f31e_redacted.txt	612d41be-3674-43e8-b755-cb2769ebb3a9.txt		9583ff38-030d-4d74-9849-2e047285b32f_redacted.txt	da103ce1-908a-45a4-ae09-999754a00a92_redacted.txt
1826ca67-cdeb-4b34-9caa-c306069fc6a1.txt		612d41be-3674-43e8-b755-cb2769ebb3a9_redacted.txt	95fdf502-e4b6-4564-b422-f9c86942b44e.txt		e1c6880a-5b1f-45a4-803c-4432b7ae1f7a.txt
1826ca67-cdeb-4b34-9caa-c306069fc6a1_redacted.txt	61698986-c731-4f2e-b3ea-37ae07ede17d.txt		95fdf502-e4b6-4564-b422-f9c86942b44e_redacted.txt	e26e9080-e60f-42b6-991c-ecbe61bc0cbd.txt
1c937a57-7406-44aa-96d8-f3da4c3b0fd8.txt		61698986-c731-4f2e-b3ea-37ae07ede17d_redacted.txt	9735d6cc-2970-4bd2-bf8f-13727d10619d.txt		e26e9080-e60f-42b6-991c-ecbe61bc0cbd_redacted.txt
1c937a57-7406-44aa-96d8-f3da4c3b0fd8_redacted.txt	61724bc9-6ef3-4120-9e70-6a170736867b.txt		9735d6cc-2970-4bd2-bf8f-13727d10619d_redacted.txt	e3dcbc0a-e5cd-4c37-8422-0804c036bef4.txt
1cbd67fb-72ce-4640-b3e7-3b76e8ef9e8e.txt		617370be-786b-4f84-b333-60f50c78e2e1.txt		97dd9bd9-6ab4-4184-a399-c7047e79e5bf.txt		e3dcbc0a-e5cd-4c37-8422-0804c036bef4_redacted.txt
1cbd67fb-72ce-4640-b3e7-3b76e8ef9e8e_redacted.txt	617370be-786b-4f84-b333-60f50c78e2e1_redacted.txt	98640ab4-9f9e-4ec6-b98d-aec97a226b54.txt		e3fdd041-5ec2-4500-a78c-80e6ddb4e5dc.txt
1d1c247e-4e99-4a02-aa95-295c9a9f0a61.txt		638db9c4-ead9-4f4a-88e8-f8b783a92c32.txt		98640ab4-9f9e-4ec6-b98d-aec97a226b54_redacted.txt	e4856f34-a670-4324-b3c0-6454b37d69fd.txt
1d1c247e-4e99-4a02-aa95-295c9a9f0a61_redacted.txt	638db9c4-ead9-4f4a-88e8-f8b783a92c32_redacted.txt	9977525d-f380-4636-bdde-3200e5220421.txt		e4856f34-a670-4324-b3c0-6454b37d69fd_redacted.txt
1d62605a-2552-45c2-8f79-cfbc93aa257f.txt		64390725-245e-4618-8a0b-e60192bfe2c1.txt		9c213609-5d6b-4602-84f9-bf2311548e3d.txt		e518d569-904d-4987-900a-54340a667a87.txt
235ad3ad-a68a-4a1f-9210-9c9dde7a2584.txt		64390725-245e-4618-8a0b-e60192bfe2c1_redacted.txt	9c213609-5d6b-4602-84f9-bf2311548e3d_redacted.txt	e518d569-904d-4987-900a-54340a667a87_redacted.txt
243fdcc3-fbe8-42a0-98d1-22ef6216e774.txt		64397887-4791-4248-ab37-7d954a1d50a7.txt		9c464f0a-73c4-490b-84fd-0ddf90cc77bb.txt		e69dd41c-5dfb-4023-abb2-bfe1273058c9.txt
26c585b3-367e-46f8-9905-50a794fea529.txt		64397887-4791-4248-ab37-7d954a1d50a7_redacted.txt	9c464f0a-73c4-490b-84fd-0ddf90cc77bb_redacted.txt	e69dd41c-5dfb-4023-abb2-bfe1273058c9_redacted.txt
26c585b3-367e-46f8-9905-50a794fea529_redacted.txt	64499b48-540a-4268-bd8d-eb402da6fcf7.txt		9de29a01-0f81-4373-9a2e-4a2fecd7f825.txt		e6c3bb53-e4ad-4621-88a6-3b517dff0718.txt
275a8a8c-a153-4549-9054-f4e3b3813f4a.txt		64499b48-540a-4268-bd8d-eb402da6fcf7_redacted.txt	9e68401e-dc52-49f0-acfa-5a71e172371b.txt		e6c3bb53-e4ad-4621-88a6-3b517dff0718_redacted.txt
29156f93-7b45-46be-afc4-5f7252d0582c.txt		6474255e-22c1-45d7-81fc-4d440c6a680d.txt		9e68401e-dc52-49f0-acfa-5a71e172371b_redacted.txt	e7bcc177-8f2f-4352-bf07-427c44cd7816.txt
29156f93-7b45-46be-afc4-5f7252d0582c_redacted.txt	6474255e-22c1-45d7-81fc-4d440c6a680d_redacted.txt	9eae6720-0263-4795-a954-b0a0c9dc2957.txt		e7ec231b-8ddd-4f65-a11f-f0373cc2f4e3.txt
29aca58e-c630-4e70-970e-eacf640d5515.txt		64f0d70a-d3aa-4112-a73c-b2d857d950bd.txt		9f00780c-a796-42bd-aa5f-3c71f243b128.txt		e7ec231b-8ddd-4f65-a11f-f0373cc2f4e3_redacted.txt
29aca58e-c630-4e70-970e-eacf640d5515_redacted.txt	64f0d70a-d3aa-4112-a73c-b2d857d950bd_redacted.txt	9f00780c-a796-42bd-aa5f-3c71f243b128_redacted.txt	e86de9bf-4cc0-4dc9-b3ea-64eae4098d85.txt
29e78334-3d37-42b7-83a5-045f07bec7b4.txt		655a856e-0e9d-470d-b49d-8b208f9b35da.txt		9f124b6a-bf8c-456a-a359-2249c1f9b378.txt		e86de9bf-4cc0-4dc9-b3ea-64eae4098d85_redacted.txt
29e78334-3d37-42b7-83a5-045f07bec7b4_redacted.txt	6763277c-cd4a-48b5-8892-ba23beb105fd.txt		9f124b6a-bf8c-456a-a359-2249c1f9b378_redacted.txt	e921172c-0612-4555-bb44-c4858f1ac445.txt
2a0269e4-bbe4-4987-b15b-f90125f5f2f4.txt		6763277c-cd4a-48b5-8892-ba23beb105fd_redacted.txt	a0dfb903-87a7-4198-bf2b-223911cb887f.txt		e921172c-0612-4555-bb44-c4858f1ac445_redacted.txt
2a0269e4-bbe4-4987-b15b-f90125f5f2f4_redacted.txt	682a2e85-35f0-4c6c-9d74-5c510a9a1611.txt		a0dfb903-87a7-4198-bf2b-223911cb887f_redacted.txt	ea513e4c-bc1f-47f4-98b8-c592336e81c0.txt
2bc9c883-5b51-4bcf-b684-c0f46043efa4.txt		685c80a5-5479-474c-97d3-6a5d31910216.txt		a1d81238-cc9f-4619-953d-28c3a5927963.txt		ea513e4c-bc1f-47f4-98b8-c592336e81c0_redacted.txt
2bc9c883-5b51-4bcf-b684-c0f46043efa4_redacted.txt	685c80a5-5479-474c-97d3-6a5d31910216_redacted.txt	a1d81238-cc9f-4619-953d-28c3a5927963_redacted.txt	eaa5804d-6153-4c4f-a390-38639d79d759.txt
2f67b402-9643-4014-af53-af5a7e6a3fea.txt		69122439-5db3-4c99-8b02-474a491a33c4.txt		a4273757-8d2e-4eb8-8198-7f53bdc6cce7.txt		eaa5804d-6153-4c4f-a390-38639d79d759_redacted.txt
30bf61d7-6e68-4285-8448-80afbb9739a1.txt		69122439-5db3-4c99-8b02-474a491a33c4_redacted.txt	a4273757-8d2e-4eb8-8198-7f53bdc6cce7_redacted.txt	eaa82b24-9df4-4cc4-9632-4578ad870828.txt
30d4cc5f-85ef-46dd-b035-dbdd2602a15f.txt		696c082d-4329-4344-9c56-1cdf255a171d.txt		a5601a82-1cbc-407a-8dbf-0e40d468412f.txt		eaa82b24-9df4-4cc4-9632-4578ad870828_redacted.txt
30d4cc5f-85ef-46dd-b035-dbdd2602a15f_redacted.txt	696c082d-4329-4344-9c56-1cdf255a171d_redacted.txt	a5601a82-1cbc-407a-8dbf-0e40d468412f_redacted.txt	eb718ec1-f1b9-4bd0-908c-a4ed9819acc4.txt
30f71de5-dc42-44dd-80e1-305c3acfb99c.txt		6a218b8a-6f63-4cbd-9516-613c6efb2b21.txt		a58e47fc-bea4-49eb-bfbd-dfdfe1535f6e.txt		eb718ec1-f1b9-4bd0-908c-a4ed9819acc4_redacted.txt
30f71de5-dc42-44dd-80e1-305c3acfb99c_redacted.txt	6d7d44e1-fd5e-4c47-8d5d-86be161f4eef.txt		a58e47fc-bea4-49eb-bfbd-dfdfe1535f6e_redacted.txt	eb937a47-7b9e-453f-9391-e8fd6c1972a0.txt
32d2fefb-daa9-4221-abdf-f2bcd42aaee7.txt		6d7d44e1-fd5e-4c47-8d5d-86be161f4eef_redacted.txt	a809f2b1-c866-4a41-a656-70d4e04beeae.txt		eb937a47-7b9e-453f-9391-e8fd6c1972a0_redacted.txt
32d2fefb-daa9-4221-abdf-f2bcd42aaee7_redacted.txt	6fb3b42c-2140-4247-af80-6baa883938f2.txt		a809f2b1-c866-4a41-a656-70d4e04beeae_redacted.txt	ec6a69ac-4f32-4be7-9da9-5b2f0e7b44a3.txt
32e94a54-7339-4455-a29f-9b61a78adc05.txt		6fb3b42c-2140-4247-af80-6baa883938f2_redacted.txt	a81d9177-1b08-47b1-99af-d5ea9e0fdc78.txt		ec6a69ac-4f32-4be7-9da9-5b2f0e7b44a3_redacted.txt
32e94a54-7339-4455-a29f-9b61a78adc05_redacted.txt	6fb50f70-d67c-4de0-9dd2-0343ba403a2d.txt		a81d9177-1b08-47b1-99af-d5ea9e0fdc78_redacted.txt	eca7006c-f5c2-43b4-9975-16414e5b8841.txt
337b4488-d596-48c2-a4fd-60d0d9cfbe42.txt		6fb50f70-d67c-4de0-9dd2-0343ba403a2d_redacted.txt	a849c222-b01d-4b90-b304-46e5c43567dc.txt		ed3bd69a-6b6a-4f71-9efb-c25ec60daf0a.txt
337b4488-d596-48c2-a4fd-60d0d9cfbe42_redacted.txt	7080f880-9ddc-4edf-95ce-efd3a3fe590c.txt		a849c222-b01d-4b90-b304-46e5c43567dc_redacted.txt	ed3bd69a-6b6a-4f71-9efb-c25ec60daf0a_redacted.txt
34b3f2b5-9630-42f1-8d6a-9810950ab88c.txt		7080f880-9ddc-4edf-95ce-efd3a3fe590c_redacted.txt	a8525c41-dc38-4ea0-857e-9a155cf878d3.txt		edf566e6-22a1-49c5-9943-cb831110ce1b.txt
34b3f2b5-9630-42f1-8d6a-9810950ab88c_redacted.txt	718313fc-a3d5-4ece-9949-4d43651478bb.txt		a8ed7a19-e891-474a-8308-20144e86df9f.txt		edf566e6-22a1-49c5-9943-cb831110ce1b_redacted.txt
356989e0-828b-4f4a-a999-4cdc5cab1820.txt		7190d0f5-1950-4b34-a48d-bc09ccbf93b0.txt		a93d19a4-a89f-438f-914b-c77f01ac9334.txt		efae32f4-cd77-4170-95a5-70e4be3405b0.txt
356989e0-828b-4f4a-a999-4cdc5cab1820_redacted.txt	71d651a9-f4c9-4982-9e01-22a8d851dddb.txt		a9a2e482-da45-45ff-bd78-654b601bcd6b.txt		efae32f4-cd77-4170-95a5-70e4be3405b0_redacted.txt
35d108c2-bf81-4b62-a5c8-5d503bf10550.txt		71d651a9-f4c9-4982-9e01-22a8d851dddb_redacted.txt	a9a2e482-da45-45ff-bd78-654b601bcd6b_redacted.txt	efb8dc67-c527-47c9-8f87-350174c301fe.txt
37089f0f-4844-4055-a3fd-864c86e70fdb.txt		71f5fcf9-aa62-46d8-ab41-01f8bc257c8a.txt		ab4b2299-66b8-4171-92dd-67f3951510e5.txt		efb8dc67-c527-47c9-8f87-350174c301fe_redacted.txt
37089f0f-4844-4055-a3fd-864c86e70fdb_redacted.txt	71f5fcf9-aa62-46d8-ab41-01f8bc257c8a_redacted.txt	ab4b2299-66b8-4171-92dd-67f3951510e5_redacted.txt	f10eee27-64a5-48f1-944c-8a2c1f3cc1cb.txt
3774289b-f28c-4325-bfd3-2398319cf80f.txt		733201ae-c38d-45f5-98fa-6d4e1b9f256f.txt		ac37ffe1-61a9-4d7f-a56b-3700368d0934.txt		f249e79c-1cf8-4666-ac8b-8b7fbc0bfaad.txt
3774289b-f28c-4325-bfd3-2398319cf80f_redacted.txt	73b2cf36-4f0e-4f2f-9dd6-d71cf62ed9c5.txt		ac37ffe1-61a9-4d7f-a56b-3700368d0934_redacted.txt	f3517462-159f-4e22-973e-4bebb5bd9179.txt
37fd4791-a4fa-42ba-a552-048185e85482.txt		7446f6a8-7b4b-45b4-bbf6-ba88fbd9179a.txt		ac6ac7f3-3837-4133-ba1c-ddee36a531e7.txt		f3517462-159f-4e22-973e-4bebb5bd9179_redacted.txt
37fd4791-a4fa-42ba-a552-048185e85482_redacted.txt	744a922f-de6e-4ee7-adbb-2475e598d991.txt		ac6ac7f3-3837-4133-ba1c-ddee36a531e7_redacted.txt	f6fcf1c3-fbbd-4c2c-8804-ed8e69ddb634.txt
38da6d6a-46b3-428d-84fa-f99f9caa866c.txt		751a55c9-e56b-4551-943e-d17c768b5c9a.txt		ac6add5e-440d-4065-bf2b-81ec7262a3b9.txt		f6fcf1c3-fbbd-4c2c-8804-ed8e69ddb634_redacted.txt
38da6d6a-46b3-428d-84fa-f99f9caa866c_redacted.txt	751a55c9-e56b-4551-943e-d17c768b5c9a_redacted.txt	ac6add5e-440d-4065-bf2b-81ec7262a3b9_redacted.txt	f7355294-243e-4848-bca0-3661e0127f77.txt
395b380e-c91d-4d93-a091-e69f3904a120.txt		75c019d4-c425-4ac3-9ae5-0265bef570f4.txt		ad070d99-770a-4b29-9b1b-719868b10abc.txt		f7355294-243e-4848-bca0-3661e0127f77_redacted.txt
395b380e-c91d-4d93-a091-e69f3904a120_redacted.txt	75c019d4-c425-4ac3-9ae5-0265bef570f4_redacted.txt	ad070d99-770a-4b29-9b1b-719868b10abc_redacted.txt	f73ecae0-bd36-4f7b-b8ad-52fbb956396f.txt
3a0ad9e4-147f-4bad-b53d-2704af0d6655.txt		75ef1f42-e8eb-4187-8cb9-5cfafeff0e09.txt		ae2575df-6159-4c1d-8ff9-8d03f8517125.txt		f73ecae0-bd36-4f7b-b8ad-52fbb956396f_redacted.txt
3a0ad9e4-147f-4bad-b53d-2704af0d6655_redacted.txt	75ef1f42-e8eb-4187-8cb9-5cfafeff0e09_redacted.txt	ae2575df-6159-4c1d-8ff9-8d03f8517125_redacted.txt	f83a341b-d043-45ca-aba3-8e229cd1e884.txt
3a116d71-0681-440e-b23c-aec81881b33f.txt		765e9ff3-7f4d-4e4c-979a-f00d24f23989.txt		af16807d-2dfb-4854-8776-06c9fc1d5111.txt		f83a341b-d043-45ca-aba3-8e229cd1e884_redacted.txt
3a116d71-0681-440e-b23c-aec81881b33f_redacted.txt	788cf216-bc1b-43c7-a508-369de58b774c.txt		af16807d-2dfb-4854-8776-06c9fc1d5111_redacted.txt	f8b77f28-1f22-4224-95d7-6f8d48911de1.txt
3a1e173c-c40f-4042-b68a-6e4d6c06daa8.txt		788cf216-bc1b-43c7-a508-369de58b774c_redacted.txt	af6c70cf-9643-4735-8ddf-eec3cbe07286.txt		f8b77f28-1f22-4224-95d7-6f8d48911de1_redacted.txt
3e1b37db-1135-45fd-990a-ea65aba5e8c7.txt		789da7f7-39f5-4326-95f6-b1ff49a5c1d1.txt		af93926a-08c3-4790-98e5-047f68834d3a.txt		f8b8070d-91e4-4758-b4ee-7187ce6fef32.txt
3e1b37db-1135-45fd-990a-ea65aba5e8c7_redacted.txt	789da7f7-39f5-4326-95f6-b1ff49a5c1d1_redacted.txt	af93926a-08c3-4790-98e5-047f68834d3a_redacted.txt	f8b8070d-91e4-4758-b4ee-7187ce6fef32_redacted.txt
3eaa09ae-ae9f-4e48-9c04-e19bb5acc2f9.txt		78c29724-5264-4f7a-81f5-414c3eba9087.txt		b0559b4e-f80a-428a-8f6b-122b3c13b505.txt		f9443e48-662a-47d9-9865-17c71452b992.txt
3eaa09ae-ae9f-4e48-9c04-e19bb5acc2f9_redacted.txt	7909388c-7065-4fa6-acaf-327983e50d49.txt		b232ee72-27b9-4896-99bd-ba45a74bd8b4.txt		f9443e48-662a-47d9-9865-17c71452b992_redacted.txt
402e436a-7b49-465b-899a-98c1a5fe44a6.txt		79927c3d-3ece-47e7-8ddd-4a6a4664dc12.txt		b232ee72-27b9-4896-99bd-ba45a74bd8b4_redacted.txt	f99e9bad-07e7-4c1e-8a2c-535c4c4a18b6.txt
402e436a-7b49-465b-899a-98c1a5fe44a6_redacted.txt	79927c3d-3ece-47e7-8ddd-4a6a4664dc12_redacted.txt	b462f058-b70d-42b3-8e96-37ea43660304.txt		f99e9bad-07e7-4c1e-8a2c-535c4c4a18b6_redacted.txt
427533c2-2eec-4f12-ac2b-ea74dcb0e5c8.txt		7aa5c474-5cab-411e-9503-af174317086a.txt		b462f058-b70d-42b3-8e96-37ea43660304_redacted.txt	fa036dc7-5059-4c21-8a4f-3c62da3c5681.txt
42e8740c-304b-4cbe-8c34-ce3110337c37.txt		7aa5c474-5cab-411e-9503-af174317086a_redacted.txt	b49694b9-862f-4700-a5f3-4290697b798c.txt		fa036dc7-5059-4c21-8a4f-3c62da3c5681_redacted.txt
42e8740c-304b-4cbe-8c34-ce3110337c37_redacted.txt	7bfc689e-15b4-428a-b20a-b4621f6901f5.txt		b49694b9-862f-4700-a5f3-4290697b798c_redacted.txt	fc7536dd-21ec-4276-891e-9ae25d55a68c.txt
434f8d6d-c3a3-4978-9660-ca0010e80061.txt		7bfc689e-15b4-428a-b20a-b4621f6901f5_redacted.txt	b4c44cd2-575d-4832-918a-e02fb7de2024.txt
44839e84-63ca-40ae-a23c-e4f145002371.txt		7d3bd193-9ecd-48e1-9408-398b36296a12.txt		b4c44cd2-575d-4832-918a-e02fb7de2024_redacted.txt
44839e84-63ca-40ae-a23c-e4f145002371_redacted.txt	7d3bd193-9ecd-48e1-9408-398b36296a12_redacted.txt	b53846fb-8956-40cd-bf5f-001e9306dbb2.txt"""

c= a.replace('\n', ' ').replace('\t', ' ').replace('.txt', '').replace('redacted', '')

"""Step 2: Run the cp command in a for loop within the terminal

for i in (print c and then paste here); do aws s3 cp  "/Users/shannon.fee/Downloads/raw_text/$i.txt" "s3://nlp-train1-us-east-1/us-west-2/processed/documents/$i/$i.extracted_text.txt"; done

"""
